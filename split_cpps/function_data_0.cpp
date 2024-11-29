class myClass {
    public:
        int publicint;

        int getPrivateInt() const { // class member that returns private variable SEF
            return privateint_;
        }
        
    private:
        int privateint_;
        mutable int mutint_; // CURRENTLY NO WAY TO CHECK IF MUTABLE WITHOUT CONTEXT
};