void register_core_extensions() {
	OS::get_singleton()->benchmark_begin_measure("Core", "Register Extensions");

	// Hardcoded for now.
	GDExtension::initialize_gdextensions();
	gdextension_manager->load_extensions();
	gdextension_manager->initialize_extensions(GDExtension::INITIALIZATION_LEVEL_CORE);
	_is_core_extensions_registered = true;

	OS::get_singleton()->benchmark_end_measure("Core", "Register Extensions");
}