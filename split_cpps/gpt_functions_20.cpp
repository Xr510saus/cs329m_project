// 6. Modifies a global array
int global_array[10] = {0};
void update_global_array(int index, int value) {
    if (index >= 0 && index < 10) {
        global_array[index] = value;
    }
}