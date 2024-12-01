#include <iostream>
#include <vector>

using namespace std;

int x = 5;
vector<int> testArray{1, 2, 3, 4, 5};
const int* x_point = &x;
const int& x_ref = x;


class myClass {
    public:
        int publicint;

        int getPrivateInt() const { // class member that returns private variable SEF
            return privateint_;
        }

        void setPrivateInt(int value) { // SE 
            privateint_ = value;
        }

        void changeMut() { // SEF
            mutint_ = 5;
        }

    private:
        int privateint_;
        mutable int mutint_; // CURRENTLY NO WAY TO CHECK IF MUTABLE WITHOUT CONTEXT
};


// SIDE EFFECT
void setFive(int& var_to_set) { // setting variable passed in by reference
    var_to_set = 5;
}

void setSix(){ // setting global variable
    x = 6; 
}

void printVar(auto var_to_print) { // printing
    cout << var_to_print << endl;
}

void squareSelf(int& sSval) {
    sSval *= sSval;
}

int squareGlobal() {
    x *= x;
    return x;
}

void incrementGlobal() {
    ++x;
}

int incrementedNotLocal() {
    x++;
    return x;
}

float unrelatedNameToo(float& u, float& v) {
    float w = (u = v);
    return w;
}

int modifyPoint(int* points) {
    *points += 1;
    return *points;
}

void freeMem(int* memory) {
    delete memory;
}

int* createMem() {
    int* placeholder = new int{42};
    return placeholder;
}

void changePoint(int* pointed, int& ref) {
    pointed = &ref;
}

void indirectRefChange(const int& cref) {
    const_cast<int&>(cref) = 5;
}

void indirectPointChange(const int* cpoint) {
    *const_cast<int*>(cpoint) = 5;
}


//SIDE EFFECT FREE
int sum(vector<int>& array_to_sum) { // returning sum of vector passed in by ref
    int sum_total = 0;

    for (int i = 0; i < array_to_sum.size(); ++i) {
        sum_total += array_to_sum[i];
    }

    return sum_total;
}

vector<int> addToArray(vector<int> array_copy, int value_to_add) { // returning modified array passed in by value
    for (int i = 0; i < array_copy.size(); ++i) {
        array_copy[i] += value_to_add;
    }
    return array_copy;
}

int squareReference(int& sRval) { // returning square of input passed in by reference
    return sRval * sRval;
}

int squareValue(int sVval) { // returning square of input passed in by value
    return sVval * sVval;
}

float unrelatedName(float u, float v) {
    float w = (u = v);
    return w;
}

int pointedInt(int* point) {
    return *point;
}

void netNothing() {
    int* temp = new int{42};
    delete temp;
}

void changeWithin(int changed) {
    changed = 5;
}

int returnChanges() {
    int rc = 10;
    rc = 5;
    return rc;
}


// MAIN TEST
int main() {
    cout << sum(testArray) << endl;
}