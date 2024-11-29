CKeyID ToKeyID(const PKHash& key_hash)
{
    return CKeyID{uint160{key_hash}};
}