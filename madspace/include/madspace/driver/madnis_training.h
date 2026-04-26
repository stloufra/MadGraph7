#pragma once

#include <chrono>

#include "madspace/compgraphs.h"
#include "madspace/driver/adam_optimizer.h"
#include "madspace/phasespace.h"

namespace madspace {

class MadnisTraining {
public:
    static void set_abort_check_function(std::function<void(void)> func) {
        _abort_check_function = func;
    }

    struct Config {
        double learning_rate = 1e-3;
        std::size_t batches = 1000;
        std::size_t log_interval = 100;
        std::size_t integration_history_length = 1000;
        std::size_t channel_dropping_interval = 100;
        double channel_dropping_threshold = 0.01;
        std::size_t cpu_generator_batch_size = 1000;
        std::size_t gpu_generator_batch_size = 64000;
        std::size_t generator_target_size_factor = 32;
        std::size_t batch_size_offset = 512;
        std::size_t batch_size_per_channel = 128;
        double uniform_channel_ratio = 0.1;
        AdamOptimizer::LRSchedule lr_schedule = AdamOptimizer::none;
        double adam_beta1 = 0.9;
        double adam_beta2 = 0.999;
        double adam_eps = 1e-8;
    };
    MadnisTraining(
        ContextPtr generator_context,
        ContextPtr optimizer_context,
        const Config& config,
        const std::vector<std::shared_ptr<Integrand>>& integrands,
        const std::optional<ChannelWeightNetwork>& cwnet
    );
    void train();

private:
    struct SampleBatch {
        std::size_t size = 0;
        std::size_t channel_index = 0;
        std::vector<std::size_t> channel_sizes;
        TensorVec tensors;
        std::size_t consumed_count = 0;
    };
    struct ChannelData {
        std::vector<std::unique_ptr<SampleBatch>> sample_batches;
        std::vector<std::tuple<std::size_t, double, double>> integration_history;
        std::size_t history_index = 0;
        std::size_t sample_count = 0;
        std::shared_ptr<Integrand> integrand;
        std::shared_ptr<IntegrandProbability> integrand_prob;
        RuntimePtr generator_runtime;
    };

    inline static std::function<void(void)> _abort_check_function = [] {};

    void build_runtimes_and_optimizer();
    std::vector<std::size_t> compute_channel_sizes();
    void start_generator_jobs(const std::vector<std::size_t>& channel_fractions);
    TensorVec permute_tensors(const TensorVec& tensors) const;
    void start_single_job(std::size_t channel_index, std::size_t batch_size);
    void start_multi_job(const std::vector<std::size_t> batch_sizes);
    bool check_training_batch(const std::vector<std::size_t>& channel_sizes);
    TensorVec build_training_batch(const std::vector<size_t>& counts);
    void process_job_results(const std::vector<std::size_t>& job_ids);
    void
    update_history(const TensorVec& results, const std::vector<std::size_t>& counts);
    void drop_channels();
    void print_progress_init();
    void print_progress_update(std::size_t batch_index);

    ContextPtr _generator_context;
    ContextPtr _optimizer_context;
    std::optional<ChannelWeightNetwork> _cwnet;
    Config _config;
    RuntimePtr _multi_channel_generator;
    std::optional<AdamOptimizer> _optimizer;
    std::vector<ChannelData> _channels;
    std::unordered_map<std::size_t, std::unique_ptr<SampleBatch>> _running_jobs;
    std::vector<double> _loss_history;
    std::size_t _loss_history_index = 0;
    std::size_t _job_id = 0;
    Tensor _generator_params;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::vector<std::size_t> _arg_permutation;
};

} // namespace madspace
