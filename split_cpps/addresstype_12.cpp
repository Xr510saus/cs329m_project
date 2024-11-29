    CScript operator()(const WitnessUnknown& id) const
    {
        return CScript() << CScript::EncodeOP_N(id.GetWitnessVersion()) << id.GetWitnessProgram();
    }