void DeserializeDB(Stream& stream, Data&& data, bool fCheckSum = true)
{
    HashVerifier verifier{stream};
    // de-serialize file header (network specific magic number) and ..
    MessageStartChars pchMsgTmp;
    verifier >> pchMsgTmp;
    // ... verify the network matches ours
    if (pchMsgTmp != Params().MessageStart()) {
        throw std::runtime_error{"Invalid network magic number"};
    }

    // de-serialize data
    verifier >> data;

    // verify checksum
    if (fCheckSum) {
        uint256 hashTmp;
        stream >> hashTmp;
        if (hashTmp != verifier.GetHash()) {
            throw std::runtime_error{"Checksum mismatch, data corrupted"};
        }
    }
}