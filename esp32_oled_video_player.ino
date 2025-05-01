#include "FS.h"
#include "SPIFFS.h"
#include "SSD1306.h"

/* You only need to format SPIFFS the first time you run a
   test or else use the SPIFFS plugin to create a partition
   https://github.com/me-no-dev/arduino-esp32fs-plugin */
#define FORMAT_SPIFFS_IF_FAILED true

SSD1306 display(0x3c, 17, 16); // ADDRESS, SDA, SCL

// width: 128, height: 64
unsigned char buf[128 * 64 / 8] = { 0 };

void readFile(fs::FS &fs, const char *path)
{
    Serial.printf("Reading file: %s\r\n", path);

    File file = fs.open(path);
    if (!file || file.isDirectory()) {
        Serial.println("- failed to open file for reading");
        return;
    }

    Serial.println("- read from file:");
    for (int j = 0; j < file.size() / 1024; j++) {
        for (int i = 0; i < sizeof(buf) / sizeof(buf[0]); i++) {
            buf[i] = file.read();
        }
        display.clear();
        display.drawXbm(0, 0, 128, 64, buf);
        display.display();
        delay(24);
    }
    file.close();
}

void setup()
{
    Serial.begin(115200);

    // Initialising the UI will init the display too.
    display.init();

    // This will make sure that multiple instances of a display driver
    // running on different ports will work together transparently
    display.setI2cAutoInit(true);

    display.flipScreenVertically();

    if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
        Serial.println("SPIFFS Mount Failed");
        return;
    }
    delay(5000);

    readFile(SPIFFS, "/video.bin");

    Serial.println("Playback complete");
}

void loop()
{
    delay(1000);
}