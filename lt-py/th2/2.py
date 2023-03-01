import datetime

# Get the time input from the user
time_str = input("Enter the time in the format 'hh:mm am/pm': ")
time_obj = datetime.datetime.strptime(time_str, '%I:%M %p')

# Get the original time zone from the user
orig_tz_str = input("Enter the original time zone offset from UTC (e.g. -08:00): ")
orig_tz = datetime.timezone(datetime.timedelta(hours=int(orig_tz_str[:3]), minutes=int(orig_tz_str[4:])))

# Get the desired time zone from the user
dest_tz_str = input("Enter the desired time zone offset from UTC (e.g. +02:30): ")
dest_tz = datetime.timezone(datetime.timedelta(hours=int(dest_tz_str[:3]), minutes=int(dest_tz_str[4:])))

# Convert the time to the original time zone
time_obj = time_obj.replace(tzinfo=orig_tz)

# Convert the time to the desired time zone
time_obj = time_obj.astimezone(dest_tz)

# Print the result
print("The converted time is:", time_obj.strftime('%I:%M %p'))
