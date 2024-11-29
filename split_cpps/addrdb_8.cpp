util::Result<std::unique_ptr<AddrMan>> LoadAddrman(const NetGroupManager& netgroupman, const ArgsManager& args)
{
    auto check_addrman = std::clamp<int32_t>(args.GetIntArg("-checkaddrman", DEFAULT_ADDRMAN_CONSISTENCY_CHECKS), 0, 1000000);
    bool deterministic = HasTestOption(args, "addrman"); // use a deterministic addrman only for tests

    auto addrman{std::make_unique<AddrMan>(netgroupman, deterministic, /*consistency_check_ratio=*/check_addrman)};

    const auto start{SteadyClock::now()};
    const auto path_addr{args.GetDataDirNet() / "peers.dat"};
    try {
        DeserializeFileDB(path_addr, *addrman);
        LogPrintf("Loaded %i addresses from peers.dat  %dms\n", addrman->Size(), Ticks<std::chrono::milliseconds>(SteadyClock::now() - start));
    } catch (const DbNotFoundError&) {
        // Addrman can be in an inconsistent state after failure, reset it
        addrman = std::make_unique<AddrMan>(netgroupman, deterministic, /*consistency_check_ratio=*/check_addrman);
        LogPrintf("Creating peers.dat because the file was not found (%s)\n", fs::quoted(fs::PathToString(path_addr)));
        DumpPeerAddresses(args, *addrman);
    } catch (const InvalidAddrManVersionError&) {
        if (!RenameOver(path_addr, (fs::path)path_addr + ".bak")) {
            return util::Error{strprintf(_("Failed to rename invalid peers.dat file. Please move or delete it and try again."))};
        }
        // Addrman can be in an inconsistent state after failure, reset it
        addrman = std::make_unique<AddrMan>(netgroupman, deterministic, /*consistency_check_ratio=*/check_addrman);
        LogPrintf("Creating new peers.dat because the file version was not compatible (%s). Original backed up to peers.dat.bak\n", fs::quoted(fs::PathToString(path_addr)));
        DumpPeerAddresses(args, *addrman);
    } catch (const std::exception& e) {
        return util::Error{strprintf(_("Invalid or corrupt peers.dat (%s). If you believe this is a bug, please report it to %s. As a workaround, you can move the file (%s) out of the way (rename, move, or delete) to have a new one created on the next start."),
                                     e.what(), CLIENT_BUGREPORT, fs::quoted(fs::PathToString(path_addr)))};
    }
    return addrman;
}