void indirectPointChange(const int* cpoint) {
    *const_cast<int*>(cpoint) = 5;
}