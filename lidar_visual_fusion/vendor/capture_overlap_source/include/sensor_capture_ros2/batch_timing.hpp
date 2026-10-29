#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sensor_capture_ros2
{

// Treat the server's calendar as UTC ONLY to measure differences. Its timezone
// and absolute clock offset cancel when anchored to the first Orin request.
inline int64_t metaTimestampNs(const std::string & json)
{
    static const std::regex pattern(
        "\\\"timestamp\\\"\\s*:\\s*\\\"([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})(?:\\.([0-9]{1,9}))?\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        throw std::invalid_argument("meta.json has no valid batch timestamp");
    }
    std::tm calendar{};
    std::istringstream input(match[1].str());
    input >> std::get_time(&calendar, "%Y-%m-%d %H:%M:%S");
    if (input.fail()) throw std::invalid_argument("invalid meta calendar");
    const auto seconds = timegm(&calendar);
    std::tm normalized{};
    gmtime_r(&seconds, &normalized);
    char rendered[32];
    std::strftime(rendered, sizeof(rendered), "%Y-%m-%d %H:%M:%S", &normalized);
    if (seconds < 0 || match[1].str() != rendered) {
        throw std::invalid_argument("invalid meta calendar date");
    }
    std::string fraction = match[2].matched ? match[2].str() : "";
    fraction.resize(9, '0');
    return static_cast<int64_t>(seconds) * 1000000000LL + std::stoll(fraction);
}

class BatchClock
{
public:
    int64_t map(int64_t meta_ns, int64_t request_ns, int64_t receive_ns)
    {
        const int64_t mapped = anchor_meta_ ? anchor_ros_ + (meta_ns - *anchor_meta_) : request_ns;
        if ((last_ros_ && mapped <= *last_ros_) || mapped > receive_ns + 2000000000LL) {
            throw std::invalid_argument("batch clock moved backwards or ahead of Orin; restart acquisition");
        }
        if (!anchor_meta_) {
            anchor_meta_ = meta_ns;
            anchor_ros_ = request_ns;
        }
        last_ros_ = mapped;
        return mapped;
    }

private:
    std::optional<int64_t> anchor_meta_;
    int64_t anchor_ros_ = 0;
    std::optional<int64_t> last_ros_;
};

inline double batchWaitSeconds(double hz, double minimum_gap, double elapsed)
{
    if (!std::isfinite(hz) || hz <= 0.0 || !std::isfinite(minimum_gap) || minimum_gap < 0.0) {
        throw std::invalid_argument("capture_freq must be positive and batch_interval nonnegative");
    }
    return std::max(minimum_gap, 1.0 / hz - elapsed);
}

}  // namespace sensor_capture_ros2
