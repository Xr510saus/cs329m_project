void ReadFromStream(AddrMan& addr, DataStream& ssPeers)
{
    DeserializeDB(ssPeers, addr, false);
}