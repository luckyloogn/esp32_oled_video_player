#include "FS.h"
#include "SPIFFS.h"
#include <Arduino.h>
#include <U8g2lib.h>

#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif

#define FORMAT_SPIFFS_IF_FAILED true

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define OLED_SCL 16 // RX2
#define OLED_SDA 17 // TX2

unsigned char buf[SCREEN_WIDTH * SCREEN_HEIGHT / 8] = { 0 };

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, OLED_SCL, OLED_SDA);

void displayCenteredText(const char *text)
{
    u8g2.clearBuffer();
    int text_width = u8g2.getStrWidth(text);
    int x = (u8g2.getDisplayWidth() - text_width) / 2;
    int y = (u8g2.getHeight() + u8g2.getAscent()) / 2;
    u8g2.drawStr(x, y, text);
    u8g2.sendBuffer();
}

void playVideo(fs::FS &fs, const char *path)
{
    File file = fs.open(path);
    if (!file || file.isDirectory()) {
        Serial.printf("Failed to open %s!", path);
        displayCenteredText("Failed to open file!");
        delay(2000);
        return;
    }

    Serial.println("Playing...");

    uint32_t fps = 0;
    file.read((uint8_t *)&fps, sizeof(fps));
    Serial.printf("Frame rate: %d.", fps);

    uint32_t delay_ms = 1000 / fps;

    while (file.available() >= sizeof(buf)) {
        for (int i = 0; i < sizeof(buf); i++) {
            buf[i] = file.read();
        }
        u8g2.clearBuffer();
        u8g2.drawXBM(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, buf);
        u8g2.sendBuffer();
        delay(delay_ms);
    }
    file.close();

    Serial.println("Playback complete.");
}

void setup()
{
    Serial.begin(115200);

    if (!u8g2.begin()) {
        Serial.println("OLED init failed!");
        while (true) {
            delay(1000);
        }
    }
    u8g2.setFont(u8g2_font_ncenB08_tr);

    displayCenteredText("OLED Video Player");

    if (!SPIFFS.begin(FORMAT_SPIFFS_IF_FAILED)) {
        Serial.println("SPIFFS Mount Failed!");
        displayCenteredText("SPIFFS Mount Failed!");
        while (true) {
            delay(1000);
        }
    }

    delay(2000);
}

void loop()
{
    playVideo(SPIFFS, "/video.data");
}
