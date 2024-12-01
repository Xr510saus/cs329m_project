#include <iostream>
#include <vector>
#include <fstream>

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
