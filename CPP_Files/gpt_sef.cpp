#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

int power(int x, int y) {
    int result = 1;
    for (int i = 0; i < y; ++i) {
        result *= x;
    }
    return result;
}

double average(const std::vector<int>& numbers) {
    if (numbers.empty()) return 0.0;
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    return static_cast<double>(sum) / numbers.size();
}

std::string reverse_string(const std::string& str) {
    std::string reversed = str;
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

bool is_palindrome(const std::string& str) {
    int left = 0;
    int right = str.length() - 1;
    while (left < right) {
        if (str[left] != str[right]) {
            return false;
        }
        ++left;
        --right;
    }
    return true;
}

bool is_positive(int x) {
    return x > 0;
}

int max_of_three(int a, int b, int c) {
    return std::max({a, b, c});
}

int count_chars(const std::string& str) {
    return str.length();
}

double distance(double x1, double y1, double x2, double y2) {
    return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
}

double median(std::vector<int> numbers) {
    if (numbers.empty()) return 0.0;
    std::sort(numbers.begin(), numbers.end());
    int n = numbers.size();
    if (n % 2 == 0) {
        return (numbers[n / 2 - 1] + numbers[n / 2]) / 2.0;
    } else {
        return numbers[n / 2];
    }
}

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.rfind(prefix, 0) == 0;
}

int reverse_number(int num) {
    int reversed = 0;
    while (num != 0) {
        reversed = reversed * 10 + num % 10;
        num /= 10;
    }
    return reversed;
}

bool is_divisible(int a, int b) {
    return b != 0 && a % b == 0;
}

int absolute_value(int x) {
    return (x < 0) ? -x : x;
}

bool is_even(int x) {
    return x % 2 == 0;
}

double area_of_circle(double radius) {
    return M_PI * radius * radius;
}

int find_max(const std::vector<int>& numbers) {
    if (numbers.empty()) return 0;  // Return 0 for an empty list
    return *std::max_element(numbers.begin(), numbers.end());
}

// 8. Opens a file and reads its content
std::string read_file_content(const std::string& filename) {
    std::ifstream file(filename);
    std::string content;
    if (file.is_open()) {
        std::getline(file, content, '\0'); // Reads the entire file content
        file.close();
    }
    return content;
}
