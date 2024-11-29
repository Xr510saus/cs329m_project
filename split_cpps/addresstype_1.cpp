CKeyID ToKeyID(const WitnessV0KeyHash& key_hash)
{
    return CKeyID{uint160{key_hash}};
}