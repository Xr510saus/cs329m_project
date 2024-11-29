vector<int> filterEvenNumbers(const vector<int>& numbers) {
    vector<int> evens;
    for (int num : numbers) {
        if (num % 2 == 0) {
            evens.push_back(num);
        }
    }
    return evens;
}