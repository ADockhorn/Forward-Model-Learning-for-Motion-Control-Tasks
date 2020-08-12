
library(msir)

files = list.files(path = "CSV", pattern = NULL, all.files = FALSE,
                   full.names = FALSE, recursive = FALSE,
                   ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)


for(file in files) {
  if (file == "out"){
    next
  }
  print(file)
  data = read.csv(file = paste("CSV/", file, sep=""), sep = " ", header=FALSE)
  l <- loess.sd(data, nsigma = 1.96)
  res = data.frame(l$x, l$y, l$sd, l$lower, l$upper)
  write.csv(res, file = paste("CSV/",'out/',file, sep=""))
}
