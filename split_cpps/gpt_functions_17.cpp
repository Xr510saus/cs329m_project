void remove_element(std::vector<int>& numbers, int value) {
    numbers.erase(std::remove(numbers.begin(), numbers.end(), value), numbers.end());
}