#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <functional>

// Strict-slot protocol: all integer fields are big endian, F32 vectors are little endian.
struct llama_remote_call {
    uint64_t sequence = 0;
    uint32_t d = 0, t = 0, k = 0;
    std::string bank, identity;
    float clamp = 0;
    bool glm = false;
    std::vector<uint32_t> pairs, ids, tokens;
    std::vector<float> input, output;
    uint64_t compute_ns = 0;
    std::array<uint32_t, 3> policies{};
    uint32_t warps = 1;
    std::string calibration;
};

class llama_remote_peer {
    int fd = -1;
    uint64_t sequence = 0;
public:
    explicit llama_remote_peer(const std::string & endpoint);
    ~llama_remote_peer();
    llama_remote_peer(const llama_remote_peer &) = delete;
    llama_remote_peer & operator=(const llama_remote_peer &) = delete;
    void send(llama_remote_call & call);
    void receive(llama_remote_call & call);
};
void llama_remote_listen(const std::string & endpoint, const std::function<void(llama_remote_call &)> & invoke);
