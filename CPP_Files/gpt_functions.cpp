#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>

using namespace std;

//SEF

int stringLength(const string &s) {
    return s.length();
}

vector<int> filterEvenNumbers(const vector<int>& numbers) {
    vector<int> evens;
    for (int num : numbers) {
        if (num % 2 == 0) {
            evens.push_back(num);
        }
    }
    return evens;
}

int factorial(int n) {
    if (n == 0) {
        return 1;
    }
    return n * factorial(n - 1);
}

bool isPalindrome(const string &s) {
    int len = s.size();
    for (int i = 0; i < len / 2; ++i) {
        if (s[i] != s[len - i - 1]) {
            return false;
        }
    }
    return true;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

bool is_prime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

double celsius_to_fahrenheit(double celsius) {
    return (celsius * 9.0 / 5.0) + 32.0;
}

//SE

void add_element(std::vector<int>& numbers, int value) {
    numbers.push_back(value);
}

void read_user_input(int& input) {
    std::cout << "Enter a number: ";
    std::cin >> input;
}

void write_to_file(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << content;
        file.close();
    }
}

int get_next_id() {
    static int id = 0;
    return ++id;
}

void clear_list(std::vector<int>& numbers) {
    numbers.clear();
}

std::unordered_map<int, std::string> database;

void update_record(int id, const std::string& new_value) {
    database[id] = new_value;
}

bool is_initialized = false;

void initialize() {
    is_initialized = true;
}

class Counter {
    public:
        void increment() {
            count++;
        }
        
        int get_count() const {
            return count;
        }

    private:
        int count = 0;
};

void remove_element(std::vector<int>& numbers, int value) {
    numbers.erase(std::remove(numbers.begin(), numbers.end(), value), numbers.end());
}

int fibonacci(int n) {
    static std::unordered_map<int, int> cache;

    if (n <= 1) return n;
    if (cache.find(n) != cache.end()) return cache[n];

    cache[n] = fibonacci(n - 1) + fibonacci(n - 2);
    return cache[n];
}

// 1. Modifies a global variable
int global_count = 0;
void increment_global_count() {
    global_count++;
}

// 6. Modifies a global array
int global_array[10] = {0};
void update_global_array(int index, int value) {
    if (index >= 0 && index < 10) {
        global_array[index] = value;
    }
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