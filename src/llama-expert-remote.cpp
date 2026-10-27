#include "llama-expert-remote.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>
#endif

namespace {
constexpr uint32_t magic = 0x45504231, version = 1;
constexpr size_t limit = 64 * 1024 * 1024;
using bytes = std::vector<uint8_t>;
void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(std::string("expert remote: ") + message); }
}
void u32(bytes & b, uint32_t x) {
    for (int i = 24; i >= 0; i -= 8) { b.push_back(uint8_t(x >> i)); }
}
uint32_t read32(const bytes & b, size_t & p) {
    require(p <= b.size() && b.size() - p >= 4, "truncated integer");
    uint32_t x = 0;
    for (int i = 0; i < 4; ++i) { x = (x << 8) | b[p++]; }
    return x;
}
void string(bytes & b, const std::string & s) {
    u32(b, s.size()); b.insert(b.end(), s.begin(), s.end());
}
std::string read_string(const bytes & b, size_t & p) {
    const size_t n = read32(b, p);
    require(n <= 256 && n <= b.size() - p, "invalid string");
    std::string s(b.begin() + p, b.begin() + p + n); p += n; return s;
}
void floats(bytes & b, const std::vector<float> & x) {
    const auto * p = reinterpret_cast<const uint8_t *>(x.data());
    if (!x.empty()) { b.insert(b.end(), p, p + x.size() * sizeof(float)); }
}
void read_floats(const bytes & b, size_t & p, std::vector<float> & x, size_t n) {
    require(n <= limit / sizeof(float) && n * sizeof(float) <= b.size() - p, "invalid vector length");
    x.resize(n); if (n) { std::memcpy(x.data(), b.data() + p, n * sizeof(float)); } p += n * sizeof(float);
}
void shape(const llama_remote_call & c) {
    require(c.d > 0 && c.d <= 16384 && c.t > 0 && c.t <= 512 && c.k > 0 && c.k <= 64, "shape exceeds limits");
    require(uint64_t(c.d) * c.t * c.k <= limit / sizeof(float), "slot buffer exceeds limit");
    require(!c.pairs.empty() && c.pairs.size() <= size_t(c.t) * c.k, "invalid pair count");
}
#ifndef _WIN32
struct socket_owner {
    int fd = -1;
    ~socket_owner() { if (fd >= 0) { ::close(fd); } }
};
sockaddr_in address(const std::string & endpoint) {
    auto colon = endpoint.find(':');
    require(colon != std::string::npos, "expected IPv4:port");
    auto ip = endpoint.substr(0, colon), port = endpoint.substr(colon + 1);
    require(!port.empty() && std::all_of(port.begin(), port.end(), [](char c) { return c >= '0' && c <= '9'; }), "invalid port");
    auto n = std::stoul(port); require(n > 0 && n <= 65535, "invalid port");
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(n);
    require(inet_pton(AF_INET, ip.c_str(), &a.sin_addr) == 1, "invalid IPv4");
    const uint32_t h = ntohl(a.sin_addr.s_addr);
    require((h >> 24) == 10 || (h >> 20) == 0xac1 || (h >> 16) == 0xc0a8, "private LAN address required");
    return a;
}
void options(int fd) {
    int one = 1; timeval timeout{30, 0};
    require(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0 &&
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0, "socket options failed");
}
bool transfer(int fd, void * data, size_t n, bool sending, bool allow_eof = false) {
    size_t done = 0;
    while (done < n) {
        auto * p = static_cast<uint8_t *>(data) + done;
        ssize_t r = sending ? ::send(fd, p, n - done, MSG_NOSIGNAL) : ::recv(fd, p, n - done, 0);
        if (r < 0 && errno == EINTR) { continue; }
        if (r == 0 && !sending && allow_eof && done == 0) { return false; }
        require(r > 0, sending ? "send failed" : "truncated reply, disconnect or timeout");
        done += size_t(r);
    }
    return true;
}
bytes header(const llama_remote_call & c, uint32_t opcode, size_t payload) {
    bytes h;
    for (uint32_t x : {magic, version, opcode, uint32_t(c.sequence >> 32), uint32_t(c.sequence), c.d, c.t, c.k,
            uint32_t(c.pairs.size()), uint32_t(c.tokens.size()), uint32_t(payload), uint32_t(c.compute_ns >> 32), uint32_t(c.compute_ns), 0u, 0u, 0u}) { u32(h, x); }
    return h;
}
void send_frame(int fd, const llama_remote_call & c, uint32_t opcode, bytes payload) {
    require(payload.size() <= limit, "message exceeds limit");
    auto frame = header(c, opcode, payload.size()); frame.insert(frame.end(), payload.begin(), payload.end());
    transfer(fd, frame.data(), frame.size(), true);
}
bool receive_frame(int fd, llama_remote_call & c, uint32_t opcode, bytes & payload) {
    bytes h(64); if (!transfer(fd, h.data(), h.size(), false, true)) { return false; }
    size_t p = 0;
    require(read32(h, p) == magic && read32(h, p) == version && read32(h, p) == opcode, "protocol mismatch");
    c.sequence = uint64_t(read32(h, p)) << 32; c.sequence |= read32(h, p);
    c.d = read32(h, p); c.t = read32(h, p); c.k = read32(h, p);
    const size_t q = read32(h, p), nt = read32(h, p), n = read32(h, p);
    c.compute_ns = uint64_t(read32(h, p)) << 32; c.compute_ns |= read32(h, p);
    require(read32(h, p) == 0 && read32(h, p) == 0 && read32(h, p) == 0, "unsupported flags");
    require(q <= 32768 && nt <= 512 && n <= limit, "frame exceeds limits");
    c.pairs.resize(q); c.tokens.resize(nt); shape(c);
    payload.resize(n); transfer(fd, payload.data(), n, false); return true;
}
#endif
}

llama_remote_peer::llama_remote_peer(const std::string & endpoint) {
#ifndef _WIN32
    const uint16_t endian = 1; require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "little-endian IEEE F32 required");
    auto a = address(endpoint); socket_owner s; s.fd = ::socket(AF_INET, SOCK_STREAM, 0); require(s.fd >= 0, "socket failed");
    options(s.fd); require(::connect(s.fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) == 0, "connect failed");
    fd = s.fd; s.fd = -1;
#else
    (void) endpoint; throw std::runtime_error("expert remote: POSIX prototype only");
#endif
}
llama_remote_peer::~llama_remote_peer() {
#ifndef _WIN32
    if (fd >= 0) { ::close(fd); }
#endif
}
void llama_remote_peer::send(llama_remote_call & c) {
#ifndef _WIN32
    shape(c); c.sequence = ++sequence;
    require(c.pairs.size() == c.ids.size() && c.input.size() == size_t(c.d) * c.tokens.size(), "invalid invocation");
    bytes b; string(b, c.bank); string(b, c.identity);
    uint32_t clamp; std::memcpy(&clamp, &c.clamp, 4); u32(b, clamp); u32(b, c.glm);
    for (size_t i = 0; i < c.pairs.size(); ++i) { u32(b, c.pairs[i]); u32(b, c.ids[i]); }
    for (auto t : c.tokens) { u32(b, t); } floats(b, c.input); send_frame(fd, c, 1, std::move(b));
#else
    (void) c;
#endif
}
void llama_remote_peer::receive(llama_remote_call & c) {
#ifndef _WIN32
    llama_remote_call reply; bytes b;
    require(receive_frame(fd, reply, 2, b), "worker disconnected");
    require(reply.sequence == c.sequence && reply.d == c.d && reply.t == c.t && reply.k == c.k && reply.pairs.size() == c.pairs.size() && reply.tokens.size() == c.tokens.size(), "reply identity mismatch");
    size_t p = 0; read_floats(b, p, c.output, size_t(c.d) * c.pairs.size()); require(p == b.size(), "reply trailing bytes"); c.compute_ns = reply.compute_ns;
#else
    (void) c;
#endif
}
void llama_remote_listen(const std::string & endpoint, const std::function<void(llama_remote_call &)> & invoke) {
#ifndef _WIN32
    const uint16_t endian = 1; require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "little-endian worker required");
    auto a = address(endpoint); socket_owner listener; listener.fd = ::socket(AF_INET, SOCK_STREAM, 0); require(listener.fd >= 0, "socket failed");
    int one = 1; setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    require(::bind(listener.fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) == 0 && ::listen(listener.fd, 1) == 0, "bind/listen failed");
    require(listener.fd < FD_SETSIZE, "listener descriptor limit");
    std::fprintf(stderr, "expert remote: listening %s\n", endpoint.c_str());
    // Context lanes share one serialized worker arena. Each lane keeps its own sequence.
    std::map<int, std::pair<std::unique_ptr<socket_owner>, uint64_t>> clients;
    for (;;) {
        fd_set ready; FD_ZERO(&ready); FD_SET(listener.fd, &ready); int largest = listener.fd;
        for (const auto & item : clients) { FD_SET(item.first, &ready); largest = std::max(largest, item.first); }
        timeval idle{60, 0}; int n = ::select(largest + 1, &ready, nullptr, nullptr, &idle);
        if (n < 0 && errno == EINTR) { continue; }
        require(n >= 0, "select failed"); if (!n) { return; }
        if (FD_ISSET(listener.fd, &ready)) {
            auto client = std::make_unique<socket_owner>(); client->fd = ::accept(listener.fd, nullptr, nullptr);
            require(client->fd >= 0 && client->fd < FD_SETSIZE && clients.size() < 16, "connection limit"); options(client->fd);
            int fd = client->fd; clients.emplace(fd, std::make_pair(std::move(client), 0));
        }
        for (auto at = clients.begin(); at != clients.end();) {
        const int fd = at->first;
        if (!FD_ISSET(fd, &ready)) { ++at; continue; }
        llama_remote_call c; bytes b;
        if (!receive_frame(fd, c, 1, b)) { at = clients.erase(at); continue; }
        require(c.sequence == ++at->second.second, "request sequence mismatch");
        size_t p = 0; c.bank = read_string(b, p); c.identity = read_string(b, p);
        uint32_t clamp = read32(b, p); std::memcpy(&c.clamp, &clamp, 4);
        const uint32_t glm = read32(b, p); require(glm <= 1, "invalid architecture policy"); c.glm = glm;
        c.ids.resize(c.pairs.size()); std::vector<uint8_t> seen(size_t(c.t) * c.k, 0), tokens(c.t, 0);
        for (size_t i = 0; i < c.pairs.size(); ++i) {
            c.pairs[i] = read32(b, p); c.ids[i] = read32(b, p);
            require(c.pairs[i] < seen.size() && !seen[c.pairs[i]]++, "duplicate or invalid slot");
        }
        for (auto & t : c.tokens) { t = read32(b, p); require(t < c.t && !tokens[t]++, "duplicate or invalid token"); }
        for (auto q : c.pairs) { require(tokens[q / c.k], "missing selected token"); }
        read_floats(b, p, c.input, size_t(c.d) * c.tokens.size()); require(p == b.size(), "request trailing bytes");
        invoke(c); require(c.output.size() == size_t(c.d) * c.pairs.size(), "invalid worker output");
        bytes result; floats(result, c.output); send_frame(fd, c, 2, std::move(result));
        ++at;
        }
    }
#else
    (void) endpoint; (void) invoke; throw std::runtime_error("expert remote: POSIX prototype only");
#endif
}
