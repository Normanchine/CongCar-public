#pragma once

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

class Logger {
private:
    std::string name_;
    std::ofstream logFile_;
    bool hasFile_ = false;

    std::string getCurrentTime() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_r(&time, &tm);
        std::ostringstream oss;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        oss << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
            << "." << std::setfill('0') << std::setw(3) << ms.count()
            << " " << name_ << "]";
        return oss.str();
    }

    void write(const std::string& msg, const char* color) {
        if (color) {
            std::cout << color << msg << "\033[0m" << std::endl;
        } else {
            std::cout << msg << std::endl;
        }
        if (hasFile_) {
            logFile_ << msg << std::endl;
        }
    }

public:
    Logger(const std::string& name, const std::string& filePath = "")
        : name_(name) {
        if (!filePath.empty()) {
            logFile_.open(filePath, std::ios::app);
            hasFile_ = logFile_.is_open();
        }
    }

    ~Logger() {
        if (hasFile_) logFile_.close();
    }

    Logger(Logger&&) = default;
    Logger& operator=(Logger&&) = default;

    template <typename... Args>
    void operator()(Args&&... args) {
        std::ostringstream oss;
        oss << getCurrentTime() << " ";
        ((oss << args << " "), ...);
        auto msg = oss.str();
        msg.pop_back(); // remove trailing space
        write(msg, nullptr);
    }

    template <typename... Args>
    void warning(Args&&... args) {
        std::ostringstream oss;
        oss << getCurrentTime() << " ";
        ((oss << args << " "), ...);
        auto msg = oss.str();
        msg.pop_back();
        write(msg, "\033[33m");
    }

    template <typename... Args>
    void error(Args&&... args) {
        std::ostringstream oss;
        oss << getCurrentTime() << " ";
        ((oss << args << " "), ...);
        auto msg = oss.str();
        msg.pop_back();
        write(msg, "\033[31m");
    }

    template <typename... Args>
    void info(Args&&... args) {
        std::ostringstream oss;
        oss << getCurrentTime() << " ";
        ((oss << args << " "), ...);
        auto msg = oss.str();
        msg.pop_back();
        write(msg, "\033[34m");
    }
};
