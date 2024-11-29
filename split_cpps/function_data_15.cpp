void indirectRefChange(const int& cref) {
    const_cast<int&>(cref) = 5;
}