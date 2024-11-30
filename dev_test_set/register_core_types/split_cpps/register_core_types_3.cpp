void register_core_singletons() {
	OS::get_singleton()->benchmark_begin_measure("Core", "Register Singletons");

	GDREGISTER_CLASS(ProjectSettings);
	GDREGISTER_ABSTRACT_CLASS(IP);
	GDREGISTER_CLASS(core_bind::Geometry2D);
	GDREGISTER_CLASS(core_bind::Geometry3D);
	GDREGISTER_CLASS(core_bind::ResourceLoader);
	GDREGISTER_CLASS(core_bind::ResourceSaver);
	GDREGISTER_CLASS(core_bind::OS);
	GDREGISTER_CLASS(core_bind::Engine);
	GDREGISTER_CLASS(core_bind::special::ClassDB);
	GDREGISTER_CLASS(core_bind::Marshalls);
	GDREGISTER_CLASS(TranslationServer);
	GDREGISTER_ABSTRACT_CLASS(Input);
	GDREGISTER_CLASS(InputMap);
	GDREGISTER_CLASS(Expression);
	GDREGISTER_CLASS(core_bind::EngineDebugger);
	GDREGISTER_CLASS(Time);

	Engine::get_singleton()->add_singleton(Engine::Singleton("ProjectSettings", ProjectSettings::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("IP", IP::get_singleton(), "IP"));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Geometry2D", core_bind::Geometry2D::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Geometry3D", core_bind::Geometry3D::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ResourceLoader", core_bind::ResourceLoader::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ResourceSaver", core_bind::ResourceSaver::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("OS", core_bind::OS::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Engine", core_bind::Engine::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ClassDB", _classdb));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Marshalls", core_bind::Marshalls::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("TranslationServer", TranslationServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Input", Input::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("InputMap", InputMap::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("EngineDebugger", core_bind::EngineDebugger::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Time", Time::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("GDExtensionManager", GDExtensionManager::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ResourceUID", ResourceUID::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("WorkerThreadPool", worker_thread_pool));

	OS::get_singleton()->benchmark_end_measure("Core", "Register Singletons");
}