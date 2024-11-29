float unrelatedNameToo(float& u, float& v) {
    float w = (u = v);
    return w;
}