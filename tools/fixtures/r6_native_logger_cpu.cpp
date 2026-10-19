// CPU-only adapter to the frozen Carbonite logger already loaded by Python.
// No Kit startup, plugin loading, graphics API or simulator calls.
#include <carb/Framework.h>
#include <carb/logging/ILogging.h>
#include <carb/logging/Log.h>
#include <cstring>

extern "C" int r6_check_native_logger(int flood)
{
    auto* framework = carb::acquireFramework("r6-cpu-logger");
    if (!framework) return 10;
    auto* logging = static_cast<carb::logging::ILogging*>(framework->acquireInterfaceWithClient(
        "r6-cpu-logger", carb::logging::ILogging::getInterfaceDesc(), nullptr));
    if (!logging) return 11;
    auto* logger = logging->getDefaultLogger();
    logger->setStandardStreamOutput(false); // Prove the marker comes from file output.
    logger->setDebugConsoleOutput(false);
    logger->setColorOutputIncluded(false);
    logging->setLogEnabled(true);
    logging->setLevelThreshold(carb::logging::kLevelVerbose);
    logger->setFileOutputLevelThreshold(carb::logging::kLevelVerbose);
    logger->setFileOuputFlushLevel(carb::logging::kLevelVerbose);
    carb::logging::LogFileConfiguration config{};
    logger->setFileConfiguration("/proc/self/fd/1", &config);
    char actual[512]{};
    auto length = logger->getFileConfiguration(actual, sizeof(actual), &config);
    if (length >= sizeof(actual) || std::strcmp(actual, "/proc/self/fd/1")) return 12;
    unsigned int count = flood ? 1000000 : 1;
    for (unsigned int i = 0; i < count; ++i)
        logging->log("R6_CPU_LOGGER", carb::logging::kLevelInfo, __FILE__, __func__, __LINE__,
                     "R6_NATIVE_FILE_MARKER_%u", i);
    logger->setFileConfiguration(nullptr, &config);
    return 0;
}
