#include "batch_timing.hpp"
#include <cassert>
#include <iostream>

using namespace sensor_capture_ros2;

int main()
{
    const auto a = metaTimestampNs(R"({"timestamp":"2026-09-15 18:34:58.352"})");
    const auto b = metaTimestampNs(R"({"timestamp":"2026-09-15 18:35:11.105"})");
    assert(b - a == 12753000000LL);
    assert(metaTimestampNs(R"({"timestamp":"2026-09-15 18:34:58"})") == a - 352000000LL);
    BatchClock clock;
    const int64_t request = a + 105000000000LL;
    assert(clock.map(a, request, request + 2000000000LL) == request);
    assert(clock.map(b, request + 13000000000LL, request + 15000000000LL) == request + b - a);
    bool backwards = false;
    try { clock.map(a, request + 16000000000LL, request + 18000000000LL); }
    catch (const std::invalid_argument &) { backwards = true; }
    assert(backwards);
    for (const auto & invalid : {"{}", R"({"timestamp":"2026-02-30 10:00:00.0"})"}) {
        bool rejected = false;
        try { metaTimestampNs(invalid); } catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    assert(batchWaitSeconds(0.25, 1.0, 2.0) == 2.0);
    assert(batchWaitSeconds(0.25, 1.0, 5.0) == 1.0);
    bool invalid_frequency = false;
    try { batchWaitSeconds(0.0, 1.0, 2.0); } catch (const std::invalid_argument &) { invalid_frequency = true; }
    assert(invalid_frequency);
    std::cout << "PASS: metadata parsing, clock alignment, backwards rejection, capture pacing\n";
}
