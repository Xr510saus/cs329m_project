int NodeMain() {
  DCHECK(base::CommandLine::InitializedForCurrentProcess());

  auto os_env = base::Environment::Create();
  bool node_options_enabled = electron::fuses::IsNodeOptionsEnabled();
  if (!node_options_enabled) {
    os_env->UnSetVar("NODE_OPTIONS");
    os_env->UnSetVar("NODE_EXTRA_CA_CERTS");
  }

#if BUILDFLAG(IS_MAC)
  if (!ProcessSignatureIsSameWithCurrentApp(getppid())) {
    // On macOS, it is forbidden to run sandboxed app with custom arguments
    // from another app, i.e. args are discarded in following call:
    //   exec("Sandboxed.app", ["--custom-args-will-be-discarded"])
    // However it is possible to bypass the restriction by abusing the node mode
    // of Electron apps:
    //   exec("Electron.app", {env: {ELECTRON_RUN_AS_NODE: "1",
    //                               NODE_OPTIONS: "--require 'bad.js'"}})
    // To prevent Electron apps from being used to work around macOS security
    // restrictions, when the parent process is not part of the app bundle, all
    // environment variables that may be used to inject scripts are removed.
    if (UnsetHijackableEnvs(os_env.get())) {
      LOG(ERROR) << "Node.js environment variables are disabled because this "
                    "process is invoked by other apps.";
    }
  }
#endif  // BUILDFLAG(IS_MAC)

#if BUILDFLAG(IS_WIN)
  v8_crashpad_support::SetUp();
#endif

#if BUILDFLAG(IS_LINUX)
  std::string fd_string, pid_string;
  if (os_env->GetVar("CRASHDUMP_SIGNAL_FD", &fd_string) &&
      os_env->GetVar("CRASHPAD_HANDLER_PID", &pid_string)) {
    int fd = -1, pid = -1;
    DCHECK(base::StringToInt(fd_string, &fd));
    DCHECK(base::StringToInt(pid_string, &pid));
    base::GlobalDescriptors::GetInstance()->Set(kCrashDumpSignal, fd);
    // Following API is unsafe in multi-threaded scenario, but at this point
    // we are still single threaded.
    os_env->UnSetVar("CRASHDUMP_SIGNAL_FD");
    os_env->UnSetVar("CRASHPAD_HANDLER_PID");
  }
#endif

  int exit_code = 1;
  {
    // Feed gin::PerIsolateData with a task runner.
    uv_loop_t* loop = uv_default_loop();
    auto uv_task_runner = base::MakeRefCounted<UvTaskRunner>(loop);
    base::SingleThreadTaskRunner::CurrentDefaultHandle handle(uv_task_runner);

    // Initialize feature list.
    auto feature_list = std::make_unique<base::FeatureList>();
    feature_list->InitFromCommandLine("", "");
    base::FeatureList::SetInstance(std::move(feature_list));

    // Explicitly register electron's builtin bindings.
    NodeBindings::RegisterBuiltinBindings();

    // Parse Node.js cli flags and strip out disallowed options.
    const std::vector<std::string> args = ElectronCommandLine::AsUtf8();
    ExitIfContainsDisallowedFlags(args);

    std::unique_ptr<node::InitializationResult> result =
        node::InitializeOncePerProcess(
            args,
            {node::ProcessInitializationFlags::kNoInitializeV8,
             node::ProcessInitializationFlags::kNoInitializeNodeV8Platform});

    for (const std::string& error : result->errors())
      fprintf(stderr, "%s: %s\n", args[0].c_str(), error.c_str());

    if (result->early_return() != 0) {
      return result->exit_code();
    }

#if BUILDFLAG(IS_LINUX)
    // On Linux, initialize crashpad after Nodejs init phase so that
    // crash and termination signal handlers can be set by the crashpad client.
    if (!pid_string.empty()) {
      auto* command_line = base::CommandLine::ForCurrentProcess();
      command_line->AppendSwitchASCII(
          crash_reporter::switches::kCrashpadHandlerPid, pid_string);
      ElectronCrashReporterClient::Create();
      crash_reporter::InitializeCrashpad(false, "node");
      crash_keys::SetCrashKeysFromCommandLine(
          *base::CommandLine::ForCurrentProcess());
      crash_keys::SetPlatformCrashKey();
      // Ensure the flags and env variable does not propagate to userland.
      command_line->RemoveSwitch(crash_reporter::switches::kCrashpadHandlerPid);
    }
#elif BUILDFLAG(IS_WIN) || (BUILDFLAG(IS_MAC) && !IS_MAS_BUILD())
    ElectronCrashReporterClient::Create();
    crash_reporter::InitializeCrashpad(false, "node");
    crash_keys::SetCrashKeysFromCommandLine(
        *base::CommandLine::ForCurrentProcess());
    crash_keys::SetPlatformCrashKey();
#endif

    gin::V8Initializer::LoadV8Snapshot(
        gin::V8SnapshotFileType::kWithAdditionalContext);

    // V8 requires a task scheduler.
    base::ThreadPoolInstance::CreateAndStartWithDefaultParams("Electron");

    // Allow Node.js to track the amount of time the event loop has spent
    // idle in the kernel’s event provider .
    uv_loop_configure(loop, UV_METRICS_IDLE_TIME);

    // Initialize gin::IsolateHolder.
    bool setup_wasm_streaming =
        node::per_process::cli_options->get_per_isolate_options()
            ->get_per_env_options()
            ->experimental_fetch;
    JavascriptEnvironment gin_env(loop, setup_wasm_streaming);

    v8::Isolate* isolate = gin_env.isolate();

    v8::Isolate::Scope isolate_scope(isolate);
    v8::Locker locker(isolate);
    node::Environment* env = nullptr;
    node::IsolateData* isolate_data = nullptr;
    {
      v8::HandleScope scope(isolate);

      isolate_data = node::CreateIsolateData(isolate, loop, gin_env.platform());
      CHECK_NE(nullptr, isolate_data);

      uint64_t env_flags = node::EnvironmentFlags::kDefaultFlags |
                           node::EnvironmentFlags::kHideConsoleWindows;
      env = node::CreateEnvironment(
          isolate_data, isolate->GetCurrentContext(), result->args(),
          result->exec_args(),
          static_cast<node::EnvironmentFlags::Flags>(env_flags));
      CHECK_NE(nullptr, env);

      node::SetIsolateUpForNode(isolate);

      gin_helper::Dictionary process(isolate, env->process_object());
      process.SetMethod("crash", &ElectronBindings::Crash);

      // Setup process.crashReporter in child node processes
      auto reporter = gin_helper::Dictionary::CreateEmpty(isolate);
      reporter.SetMethod("getParameters", &GetParameters);
#if IS_MAS_BUILD()
      reporter.SetMethod("addExtraParameter", &SetCrashKeyStub);
      reporter.SetMethod("removeExtraParameter", &ClearCrashKeyStub);
#else
      reporter.SetMethod("addExtraParameter",
                         &electron::crash_keys::SetCrashKey);
      reporter.SetMethod("removeExtraParameter",
                         &electron::crash_keys::ClearCrashKey);
#endif

      process.Set("crashReporter", reporter);
    }

    v8::HandleScope scope(isolate);
    node::LoadEnvironment(env, node::StartExecutionCallback{}, &OnNodePreload);

    // Potential reasons we get Nothing here may include: the env
    // is stopping, or the user hooks process.emit('exit').
    exit_code = node::SpinEventLoop(env).FromMaybe(1);

    node::ResetStdio();

    node::Stop(env, node::StopFlags::kDoNotTerminateIsolate);

    node::FreeEnvironment(env);
    node::FreeIsolateData(isolate_data);
  }

  // According to "src/gin/shell/gin_main.cc":
  //
  // gin::IsolateHolder waits for tasks running in ThreadPool in its
  // destructor and thus must be destroyed before ThreadPool starts skipping
  // CONTINUE_ON_SHUTDOWN tasks.
  base::ThreadPoolInstance::Get()->Shutdown();

  v8::V8::Dispose();

  return exit_code;
}