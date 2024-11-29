vector<int> addToArray(vector<int> array_copy, int value_to_add) { // returning modified array passed in by value
    for (int i = 0; i < array_copy.size(); ++i) {
        array_copy[i] += value_to_add;
    }
    return array_copy;
}