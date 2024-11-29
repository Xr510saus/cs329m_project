    CScript operator()(const WitnessV1Taproot& tap) const
    {
        return CScript() << OP_1 << ToByteVector(tap);
    }