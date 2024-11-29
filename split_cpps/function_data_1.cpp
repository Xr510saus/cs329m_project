class myClass {
    public:
        int publicint;

        void setPrivateInt(int value) { // SE 
            privateint_ = value;
        }

    private:
        int privateint_;
        mutable int mutint_; // CURRENTLY NO WAY TO CHECK IF MUTABLE WITHOUT CONTEXT
};