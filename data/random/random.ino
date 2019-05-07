int count = 0;unsigned short Xvals[150]; unsigned short Yvals[150];unsigned short Zvals[150];
unsigned long tvals[150];
void setup(){Serial.begin(115200);}
void loop(){Xvals[count]=analogRead(A0); Yvals[count]=analogRead(A1);Zvals[count]=analogRead(A2); tvals[count]=millis(); count = count+1;if(count == 150) {for (int i=0; i<150; i++){Serial.println(Xvals[i]); Serial.println(Yvals[i]);Serial.println(Zvals[i]); Serial.println(tvals[i]); } count=0;} delay(2.378779);}