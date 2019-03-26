SET(${PROJECT_NAME}_EXAMPLE_LOGS
  bipedal_walking_from_foot_traj
  jump
  quadrupedal_walking_balancing
  salto
  talos_arm
  walk
  )

# Create 2 tests: one that checks the example successfuly returns, and a second that compares its logs with older logs
# that are known to work.
MACRO(CHECK_LOG NAME)
  SET(RESULT "${CMAKE_CURRENT_SOURCE_DIR}/log/${NAME}")
  FILE(RELATIVE_PATH DIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  ADD_PYTHON_UNIT_TEST("py-eg-${NAME}" "${DIR}/${NAME}.py" ".")
  ADD_TEST(NAME "py-lg-${NAME}" COMMAND ${CMAKE_COMMAND} -E compare_files "${RESULT}.log" "${RESULT}.out")
  SET_TESTS_PROPERTIES("py-lg-${NAME}" PROPERTIES DEPENDS "py-eg-${NAME}" REQUIRED_FILES "${RESULT}.out")
ENDMACRO(CHECK_LOG NAME)

FOREACH(EXAMPLE ${${PROJECT_NAME}_EXAMPLE_LOGS})
  CHECK_LOG(${EXAMPLE})
ENDFOREACH(EXAMPLE ${${PROJECT_NAME}_EXAMPLE_LOGS})

ADD_SUBDIRECTORY(locomotion)
