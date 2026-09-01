// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <limits>

#include "paged_attention_gpu_test.h"
#include "test_utils/test_data/paged_attention_token_type_test_data.h"

// Bidirectional image mask in the PagedAttention MIXED stage (micro SDPA).
//
// smoke_paged_attention_token_type_micro_sdpa_mixed already replays the PREFILL golden data as a
// chunked prefill, but it deliberately picks a split whose scheduled chunk holds text tokens only.
// A bidirectional pair needs both the query and the key to be image tokens, so a text-only chunk is
// masked purely causally and never exercises the rule.
//
// This suite picks the opposite split: past_len stops right before the first image token, so a whole
// image group lands inside the scheduled chunk and the bidirectional rule has to fire.
//
//        |<-- past_len -->|<--------- scheduled chunk --------->|
//   type | 0  0  0        | 1  1  1  1  1  1  1  1  1  1  0  0  |
//                           ^-------- image group --------^
//                           these queries must attend to later keys of the same group
//
// Keeping the group fully inside the chunk matters: token_type_ids describes the scheduled tokens
// only, so a group straddling the chunk boundary cannot be resolved by any kernel.
//
// The suite carries its own negative control. Before touching the GPU it recomputes the sequence
// with a purely causal mask and asserts that the result disagrees with the golden data on image
// rows - otherwise the case would pass even with the bidirectional mask missing. On text rows the
// two must agree, which validates the reference implementation itself.
//
// The failure message reports how many image rows versus text rows are wrong; image rows differing
// while text rows match points at the bidirectional mask.

#ifdef ENABLE_ONEDNN_FOR_GPU

namespace {

struct mixed_bidir_test_params : public paged_attention_test_params {
    test::TestData token_type_test_data;
};

// Attention over the whole sequence with a plain causal (+ sliding window) mask - exactly what the
// MIXED kernels produce when token_type_ids is ignored. Used as a negative control below.
std::vector<float> causal_only_reference(const mixed_bidir_test_params& p) {
    const auto& data = p.token_type_test_data;
    const size_t seq_len = data.tokenTypes.size();
    const size_t heads = static_cast<size_t>(p.num_heads);
    const size_t head_size = static_cast<size_t>(p.k_head_size);
    const size_t hidden = heads * head_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const size_t window = p.sliding_window_size > 0 ? static_cast<size_t>(p.sliding_window_size) : 0;

    std::vector<float> out(seq_len * hidden, 0.0f);
    std::vector<float> weights(seq_len);

    for (size_t h = 0; h < heads; h++) {
        const size_t off = h * head_size;
        for (size_t q = 0; q < seq_len; q++) {
            const size_t first = (window > 0 && q + 1 > window) ? q + 1 - window : 0;
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t k = first; k <= q; k++) {
                float dot = 0.0f;
                for (size_t d = 0; d < head_size; d++) {
                    dot += data.qData[q * hidden + off + d] * data.kData[k * hidden + off + d];
                }
                weights[k] = dot * scale;
                max_logit = std::max(max_logit, weights[k]);
            }
            float sum = 0.0f;
            for (size_t k = first; k <= q; k++) {
                weights[k] = std::exp(weights[k] - max_logit);
                sum += weights[k];
            }
            for (size_t k = first; k <= q; k++) {
                const float w = weights[k] / sum;
                for (size_t d = 0; d < head_size; d++) {
                    out[q * hidden + off + d] += w * data.vData[k * hidden + off + d];
                }
            }
        }
    }
    return out;
}

class paged_attention_token_type_mixed_bidir_test : public PagedAttentionTest<mixed_bidir_test_params> {
public:
    // Key/Value cover the whole sequence: PagedAttentionManager preloads the leading past_len
    // tokens into the KV cache and submits the rest through the key/value inputs. Query covers the
    // scheduled chunk only.
    void apply_mixed_test_data(PagedAttentionManager& pam, const mixed_bidir_test_params& p) {
        const auto& data = p.token_type_test_data;
        ASSERT_EQ(p.subsequences.size(), 1u);

        const size_t past_len = static_cast<size_t>(p.subsequences[0].past_len);
        const size_t num_tokens = static_cast<size_t>(p.subsequences[0].num_tokens);
        const size_t seq_len = data.tokenTypes.size();
        const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.k_head_size);

        ASSERT_GT(past_len, 0u);
        ASSERT_EQ(past_len + num_tokens, seq_len);
        ASSERT_EQ(data.qData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.kData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.vData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.expectedOutput.size(), seq_len * hidden_dim);

        pam.key_data = {to_float16(data.kData)};
        pam.value_data = {to_float16(data.vData)};

        const auto query_data = to_float16(data.qData);
        pam.query_data = {std::vector<ov::float16>(query_data.begin() + past_len * hidden_dim, query_data.end())};

        pam.token_type_ids.assign(data.tokenTypes.begin() + past_len, data.tokenTypes.end());
    }

    void compare_chunk_output(cldnn::memory::ptr output_mem, const mixed_bidir_test_params& p) {
        const auto& data = p.token_type_test_data;
        const size_t past_len = static_cast<size_t>(p.subsequences[0].past_len);
        const size_t num_tokens = static_cast<size_t>(p.subsequences[0].num_tokens);
        const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.v_head_size);

        ASSERT_TRUE(output_mem);
        ASSERT_EQ(output_mem->count(), num_tokens * hidden_dim);

        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(output_mem, tests::get_test_stream());
        constexpr float tolerance = 1e-2f;

        std::vector<size_t> wrong_image_rows;
        std::vector<size_t> wrong_text_rows;
        float worst_diff = 0.0f;

        for (size_t row = 0; row < num_tokens; row++) {
            const size_t global_row = past_len + row;
            bool row_ok = true;
            for (size_t col = 0; col < hidden_dim; col++) {
                const float actual = static_cast<float>(mem_ptr[row * hidden_dim + col]);
                const float expected = data.expectedOutput[global_row * hidden_dim + col];
                const float diff = std::fabs(actual - expected);
                worst_diff = std::max(worst_diff, diff);
                if (diff > tolerance) {
                    row_ok = false;
                }
            }
            if (!row_ok) {
                auto& bucket = data.tokenTypes[global_row] == 1 ? wrong_image_rows : wrong_text_rows;
                bucket.push_back(global_row);
            }
        }

        auto format_rows = [](const std::vector<size_t>& rows) {
            std::string out;
            for (size_t i = 0; i < std::min<size_t>(rows.size(), 8); i++) {
                out += (i ? ", " : "") + std::to_string(rows[i]);
            }
            if (rows.size() > 8) {
                out += ", ...";
            }
            return out.empty() ? std::string("none") : out;
        };

        EXPECT_TRUE(wrong_image_rows.empty() && wrong_text_rows.empty())
            << "chunk output does not match the single-shot golden data.\n"
            << "  past_len            = " << past_len << ", chunk = " << num_tokens << " tokens\n"
            << "  wrong image rows    = " << wrong_image_rows.size() << " [" << format_rows(wrong_image_rows) << "]\n"
            << "  wrong text rows     = " << wrong_text_rows.size() << " [" << format_rows(wrong_text_rows) << "]\n"
            << "  worst abs diff      = " << worst_diff << " (tolerance " << tolerance << ")\n"
            << "  image rows differing while text rows match indicates the bidirectional mask was not applied.";
    }

    // Guards the suite against silently passing if the golden data ever stops depending on the
    // bidirectional rule: a purely causal reference must disagree with it on image rows.
    void assert_case_discriminates(const mixed_bidir_test_params& p) {
        const auto& data = p.token_type_test_data;
        const size_t past_len = static_cast<size_t>(p.subsequences[0].past_len);
        const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.v_head_size);
        const auto causal = causal_only_reference(p);
        constexpr float tolerance = 1e-2f;

        size_t image_rows_needing_bidir = 0;
        float worst_diff = 0.0f;
        float worst_text_diff = 0.0f;
        for (size_t row = past_len; row < data.tokenTypes.size(); row++) {
            float row_diff = 0.0f;
            for (size_t col = 0; col < hidden_dim; col++) {
                row_diff = std::max(row_diff,
                                    std::fabs(causal[row * hidden_dim + col] - data.expectedOutput[row * hidden_dim + col]));
            }
            if (data.tokenTypes[row] != 1) {
                worst_text_diff = std::max(worst_text_diff, row_diff);
                continue;
            }
            worst_diff = std::max(worst_diff, row_diff);
            if (row_diff > tolerance)
                image_rows_needing_bidir++;
        }

        // Sanity check on the reference itself: a text query is never part of a bidirectional pair,
        // so without a sliding window its row must already match the golden data.
        if (p.sliding_window_size == 0) {
            ASSERT_LE(worst_text_diff, tolerance)
                << "the causal reference disagrees with the golden data on text rows, so it cannot be "
                   "trusted as a negative control (worst abs diff " << worst_text_diff << ")";
        }

        ASSERT_GT(image_rows_needing_bidir, 0u)
            << "negative control failed: a purely causal mask already reproduces the golden data, so "
               "this case cannot detect a missing bidirectional mask.\n"
            << "  past_len       = " << past_len << ", sliding window = " << p.sliding_window_size << "\n"
            << "  worst abs diff = " << worst_diff << " (tolerance " << tolerance << ")";
    }
};

TEST_P(paged_attention_token_type_mixed_bidir_test, image_group_inside_chunk) {
    const auto& device_info = tests::get_test_engine().get_device_info();
    if (!device_info.supports_immad || device_info.arch < cldnn::gpu_arch::xe_hpg)
        GTEST_SKIP() << "Micro SDPA requires DPAS/XMX support";
    if (device_info.arch == cldnn::gpu_arch::xe3p)
        GTEST_SKIP() << "Micro SDPA is disabled on xe3p";

    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    assert_case_discriminates(p);

    apply_mixed_test_data(pam, p);
    auto result = run_gpu_inference(pam, p);

    auto pa_inst = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa_inst, nullptr);
    auto* impl = pa_inst->get_impl();
    ASSERT_NE(impl, nullptr);
    const auto kernel_entries = impl->get_kernels_dump_info(*pa_inst->get_impl_params()).get_entries();
    EXPECT_NE(kernel_entries.find("sdpa_micro"), std::string::npos) << "Expected micro SDPA kernel for MIXED, got: " << kernel_entries;

    compare_chunk_output(result.outputs.at("output_data").get_memory(), p);
}

// Stops past_len right before the first image token so the first image group is fully contained in
// the scheduled chunk. Returns 0 when no such split exists - the sequence starts with an image
// token, has no image tokens at all, has a single isolated image token (no later key to reach), or
// the resulting chunk would be short enough to be classified as GENERATE instead of MIXED.
static int find_bidir_mixed_split(const test::TestData& data) {
    const int seq_len = static_cast<int>(data.tokenTypes.size());
    int first_image_token = -1;
    int last_image_token = -1;
    for (int i = 0; i < seq_len; i++) {
        if (data.tokenTypes[i] != 1)
            continue;
        if (first_image_token < 0)
            first_image_token = i;
        last_image_token = i;
    }

    if (first_image_token <= 0 || last_image_token <= first_image_token)
        return 0;

    const int past_len = first_image_token;
    return (seq_len - past_len >= 2) ? past_len : 0;
}

static mixed_bidir_test_params make_bidir_mixed_test_param(const test::TestData& data, int past_len) {
    mixed_bidir_test_params p;
    p.subsequences = {{static_cast<int>(data.tokenTypes.size()) - past_len, past_len}};
    p.num_heads = 1;
    p.num_kv_heads = 1;
    p.k_head_size = 32;
    p.v_head_size = 32;
    p.block_size = 16;
    p.sliding_window_size = data.slidingWindowSize;
    p.kv_cache_compression = DISABLE_CACHE_COMPRESSION;
    p.key_cache_quant_mode = ov::internal::CacheQuantMode::BY_TOKEN;
    p.dynamic_paddings = STATIC_INPUT_PAD;
    p.scores_mode = DISABLE_SCORES;
    p.rotation_config = DISABLE_ROTATION;
    p.disable_flashattn_v2 = ENABLE_FA_V2;
    p.token_type_ids = std::vector<int>(data.tokenTypes.begin() + past_len, data.tokenTypes.end());
    p.token_type_test_data = data;
    return p;
}

static std::vector<mixed_bidir_test_params> make_bidir_mixed_test_params() {
    std::vector<mixed_bidir_test_params> params;
    for (const auto& data : test::PagedAttentionTokenTypeTestData::GetTestData()) {
        const int past_len = find_bidir_mixed_split(data);
        if (past_len == 0)
            continue;
        params.push_back(make_bidir_mixed_test_param(data, past_len));
    }
    return params;
}

static std::string get_bidir_mixed_test_name(const testing::TestParamInfo<mixed_bidir_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size) + "_Past" + std::to_string(p.subsequences[0].past_len) +
           "_MicroSDPA_Mixed_Bidir";
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_mixed_bidir,
                         paged_attention_token_type_mixed_bidir_test,
                         ::testing::ValuesIn(make_bidir_mixed_test_params()),
                         get_bidir_mixed_test_name);

}  // namespace

#endif  // ENABLE_ONEDNN_FOR_GPU
