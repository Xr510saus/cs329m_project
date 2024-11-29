bool DumpPeerAddresses(const ArgsManager& args, const AddrMan& addr)
{
    const auto pathAddr = args.GetDataDirNet() / "peers.dat";
    return SerializeFileDB("peers", pathAddr, addr);
}