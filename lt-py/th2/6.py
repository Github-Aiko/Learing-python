
def subtract_time(time1, time2):
    # Chuyển đổi giá trị thời gian về định dạng 24h
    time1 = convert_to_24h(time1)
    time2 = convert_to_24h(time2)

    # Tính toán khoảng thời gian giữa hai giá trị thời gian
    diff_minutes = (time2[0] * 60 + time2[1]) - (time1[0] * 60 + time1[1])

    # Chuyển đổi kết quả về định dạng h:m am/pm
    diff_hours = diff_minutes // 60
    diff_minutes = diff_minutes % 60
    if diff_hours >= 12:
        diff_hours -= 12
        suffix = "pm"
    else:
        suffix = "am"
    if diff_hours == 0:
        diff_hours = 12
    diff_time = "{:d}:{:02d} {}".format(diff_hours, diff_minutes, suffix)
    return diff_time


def convert_to_24h(time_str):
    # Chuyển đổi giá trị thời gian về định dạng 24h
    time_suffix = time_str[-2:]
    time_str = time_str[:-2]
    time_parts = time_str.split(":")
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    if time_suffix == "pm" and hours != 12:
        hours += 12
    if time_suffix == "am" and hours == 12:
        hours = 0
    return (hours, minutes)

time1 = "9:30 am"
time2 = "2:45 pm"
diff_time = subtract_time(time1, time2)
print(diff_time)  # Kết quả: 5:15 pm
