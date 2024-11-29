class myClass {
    public:
        int publicint;

        void changeMut() { // SEF
            mutint_ = 5;
        }

    private:
        int privateint_;
        mutable int mutint_; // CURRENTLY NO WAY TO CHECK IF MUTABLE WITHOUT CONTEXT
};