import datetime as dt


def as_giga_mega_kilo_bytes(bytes: int) -> str:
    return "{}GB {}MB {}kB {}B".format(int(bytes % (1024 ** 4) / 1024 ** 3), int(bytes % (1024 ** 3) / 1024 ** 2),
                                       int(bytes % (1024 ** 2) / 1024), int(bytes % 1024))


def as_h_m_s(time: dt.timedelta):
    return "{}H{:2}M{:2}S".format(time.days * 24 + (time.seconds // 3600),  # Hours
                                  (time.seconds // 60) % 60,  # Minutes
                                  time.seconds % 60)  # Seconds
