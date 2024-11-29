    CScript operator()(const PubKeyDestination& dest) const
    {
        return CScript() << ToByteVector(dest.GetPubKey()) << OP_CHECKSIG;
    }