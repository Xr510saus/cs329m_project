int sum(vector<int>& array_to_sum) { // returning sum of vector passed in by ref
    int sum_total = 0;

    for (int i = 0; i < array_to_sum.size(); ++i) {
        sum_total += array_to_sum[i];
    }

    return sum_total;
}