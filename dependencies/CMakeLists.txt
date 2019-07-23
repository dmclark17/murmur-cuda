
# Exclude from all to prevent installs in dependencies
add_subdirectory("smhasher/src" EXCLUDE_FROM_ALL)

set(depend_src dependencies/smhasher/src PARENT_SCOPE)
