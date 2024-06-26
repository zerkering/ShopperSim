library(fitdistrplus)
library(readxl)
library(logitnorm)
data <- read_excel("/Users/korrawee/Desktop/Retail Customer Behavior Project/shoppersim/statistic_result_clean.xlsx")
data <- na.omit(data)
int_arrival <- diff(data$arrive_time)
x_s<- data$shopping_time
x_a<- data$x_a
x_b<- data$x_b


#For x_s
par(mfrow = c(1,1))
descdist(x_s, boot = 1000)
plotdist(x_s, histo = TRUE, demp = TRUE)
fg <- fitdist(x_s, "gamma")
summary(fg)
fln <- fitdist(x_s, "lnorm")
summary(fln)
fw <- fitdist(x_s,'weibull')
summary(fw)
par(mfrow = c(2,2))
plot.legend <- c("lognormal")
denscomp(list(fln), legendtext = plot.legend)
qqcomp(list(fln), legendtext = plot.legend)
cdfcomp(list(fln), legendtext = plot.legend)
ppcomp(list(fln), legendtext = plot.legend)

#plot.legend <- c("gamma", "lognormal", "weibull")
#denscomp(list(fg,fln,fw), legendtext = plot.legend)
#qqcomp(list(fg,fln,fw), legendtext = plot.legend)
#cdfcomp(list(fg,fln,fw), legendtext = plot.legend)
#ppcomp(list(fg,fln,fw), legendtext = plot.legend)

# Do goodness of fit tests
#gofstat(list(fw, fln, fg),
        #fitnames = c("Weibull", "lognormal", "gamma"))

#gofstat(list(fl),fitnames =  c("Lognormal"))

#For int_arrival
par(mfrow = c(1,1))
descdist(int_arrival, boot = 1000)
plotdist(int_arrival, histo = TRUE, demp = TRUE)
fe <- fitdist(int_arrival, "exp")
summary(int_arrival)
par(mfrow = c(2,2))
plot.legend <- c("Exponential")
denscomp(list(fe), legendtext = plot.legend)
qqcomp(list(fe), legendtext = plot.legend)
cdfcomp(list(fe), legendtext = plot.legend)
ppcomp(list(fe), legendtext = plot.legend)

#For x_b
chisq.test(x_b)
par(mfrow = c(1,1))
descdist(x_b, boot = 1000)
plotdist(x_b, histo = TRUE, demp = TRUE)

#For x_a

logit_transform<- logit(x_a)
par(mfrow = c(1,1))
descdist(logit_transform, boot = 1000)
plotdist(logit_transform, histo = TRUE, demp = TRUE)
par(mfrow = c(2,2))
fn <- fitdist(logit_transform, "norm")
summary(fn)
plot.legend <- c("Normal")
denscomp(list(fn), legendtext = plot.legend)
qqcomp(list(fn), legendtext = plot.legend)
cdfcomp(list(fn), legendtext = plot.legend)
ppcomp(list(fn), legendtext = plot.legend)

par(mfrow = c(1,1))
plot(x_s, x_b, main = "Shopping Time vs Basket Size",xlab="Shopping Time", ylab="Basket Size")
plot(x_s, x_a, main = "Shopping Time vs Area Covered", xlab = "Shopping Time", ylab="Area Covered")
plot(x_b,x_a, main = "Basket Size vs Area Covered", xlab = "Basket Size", ylab="Area Covered")



