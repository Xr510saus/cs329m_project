int get_next_id() {
    static int id = 0;
    return ++id;
}