""" Functions given a day as a number
given a month as a number
return the string with a leading zero
"""
import calendar

def day_string(day):
  if day < 10:
    day_str = '0'+str(day)
  else:
    day_str = str(day)
  return day_str

def mon_string(month):
  if month < 10:
    mon_str = '0'+str(month)
  else:
    mon_str = str(month)
  return mon_str

def dayinmo(year,month):
  lastday = calendar.monthrange(year,month)[1]
  return lastday
