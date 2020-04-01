#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:21:01 2020

@author: GaborSarosi
"""

def gimatria_of(letter):
    switcher = {
            "aleph": 1,
            "beth": 2,
            "gimel": 3,
            "daleth": 4,
            "hey": 5,
            "vav": 6,
            "zayin": 7,
            "chet": 8,
            "tet": 9,
            "yud": 10,
            "chaf": 20,
            "lamed": 30,
            "mem": 40,
            "nun": 50,
            "samech": 60,
            "ayin": 70,
            "pe": 80,
            "tzadi": 90,
            "quf": 100,
            "resh": 200,
            "shin": 300,
            "tav": 400,
            "chaf_sofit": 500,
            "mem_sofit": 600,
            "nun_sofit": 700,
            "pe_sofit": 800,
            "tzadi_sofit": 900,
            "pas": 1001,
            "paspas": 1002
        }
    return switcher.get(letter, "0")


def full_name_of(character):
    switcher = {
            1: "aleph",
            2: "beth",
            3: "gimel",
            4: "daleth",
            5: "hey",
            6: "vav",
            7: "zayin",
            8: "chet",
            9: "tet",
            10: "yud",
            20: "chaf",
            30: "lamed",
            40: "mem",
            50: "nun",
            60: "samech",
            70: "ayin",
            80: "pe",
            90: "tzadi",
            100: "quf",
            200: "resh",
            300: "shin",
            400: "tav",
            500: "chaf_sofit",
            600: "mem_sofit",
            700: "nun_sofit",
            800: "pe_sofit",
            900: "tzadi_sofit",
            1001: "pas",
            1002: "paspas"
    }
    return switcher.get(character, "unknown")