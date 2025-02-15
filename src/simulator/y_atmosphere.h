#ifndef Y_ATMOSPHERE_H_
#define Y_ATMOSPHERE_H_
#include <cmath>//.....................................fabs(),exp(),pow(),sqrt()
namespace yAtmosphere{//<======================================ATMOSPHERIC MODEL
 //@@@@@@@@@@@@@@@ [BASED ON "U.S. STANDARD ATMOSPHERE, 1976"] @@@@@@@@@@@@@@@@
 const double TABLE4[8][4]={//<===============================TRANSITION POINTS
 00000 , -0.0065 , 288.150 , 1.01325000000000E+5 ,// FOR PRESSURE &
 11000 , 0.0000 , 216.650 , 2.26320639734629E+4 ,// TEMPERATURE VS
 20000 , 0.0010 , 216.650 , 5.47488866967777E+3 ,// GEOPOTENTIAL
 32000 , 0.0028 , 228.650 , 8.68018684755228E+2 ,// ALTITUDE CURVES
 47000 , 0.0000 , 270.650 , 1.10906305554966E+2 ,// [table 4]
 51000 , -0.0028 , 270.650 , 6.69388731186873E+1 ,// (3RD COLUMN IS
 71000 , -0.0020 , 214.650 , 3.95642042804073E+0 ,// TEMPERATURE,
 84852 , 0.0000 , 186.946 , 3.73383589976215E-1 // 4TH, PRESSURE)
 };//~~~YAGENAUT@GMAIL.COM~~~~~~~~~~~~~~~~~~~~~~~~~LAST~UPDATED~15MAR2013~~~~~~
 inline double Temperature(//<==================================TEMPERATURE (K)
 double z){//<--------ALTITUDE (m) (T IS VALID FOR -5,000 m < z < 86,000 m)
 double H=z*6356766/(z+6356766);//..............................[equation 18]
 int b;/*<-*/for(b=0;b<7;++b)if(H<TABLE4[b+1][0])break;
 return TABLE4[b][2]+TABLE4[b][1]*(H-TABLE4[b][0]);//...........[equation 23]
 }//~~~~YAGENAUT@GMAIL.COM~~~~~~~~~~~~~~~~~~~~~~~~~LAST~UPDATED~15MAR2013~~~~~~
 inline double Pressure(//<=======================================PRESSURE (Pa)
 double z){//<--------ALTITUDE (m) (P IS VALID FOR -5,000 m < z < 86,000 m)
 double H=z*6356766/(z+6356766);//..............................[equation 18]
 int b;/*<-*/for(b=0;b<7;++b)if(H<TABLE4[b+1][0])break;
 double C=-.0341631947363104;//................C = -G0*M0/RSTAR [pages 8,9,3]
 double Hb=TABLE4[b][0],Lb=TABLE4[b][1],Tb=TABLE4[b][2],Pb=TABLE4[b][3];
 return Pb*(fabs(Lb)>1E-12?pow(1+Lb/Tb*(H-Hb),C/Lb):exp(C*(H-Hb)/Tb));
 }//~~~~YAGENAUT@GMAIL.COM~~~~~~~~~~~~~~~~~~~~~~~~~LAST~UPDATED~15MAR2013~~~~~~
 inline double Density(//<=====================================DENSITY (kg/m^3)
 double T,//<-------------TEMPERATURE (K) (CALCULATE T USING Temperature())
 double P){//<-----------------PRESSURE (Pa) (CALCULATE P USING Pressure())
 return P*.00348367635597379/T;//...............................[equation 42]
 }//~~~~YAGENAUT@GMAIL.COM~~~~~~~~~~~~~~~~~~~~~~~~~LAST~UPDATED~15MAR2013~~~~~~
 inline double SpeedofSound(//<============================SPEED OF SOUND (m/s)
 double T){//<------------TEMPERATURE (K) (CALCULATE T USING Temperature())
 return sqrt(401.87430086589*T);//..............................[equation 50]
 }//~~~~YAGENAUT@GMAIL.COM~~~~~~~~~~~~~~~~~~~~~~~~~LAST~UPDATED~15MAR2013~~~~~~
 inline double Gravity(//<==================ACCELERATION DUE TO GRAVITY (m/s^2)
 double z){//<--------ALTITUDE (m) (g IS VALID FOR -5,000 m < z < 86,000 m)
 return 9.80665*pow(1+z/6356766,-2);//..........................[equation 17]
 }//~~~~YAGENAUT@GMAIL.COM~~~~~~~~~~~~~~~~~~~~~~~~~LAST~UPDATED~15MAR2013~~~~~~
}//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#endif