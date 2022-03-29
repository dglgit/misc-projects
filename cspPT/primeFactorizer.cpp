#include<iostream>
#include<stdio.h>
#include<string>
#include<vector>
#include<math>
using namespace std;

string nChar(int n, char c){
   char ret[n];
   for(int i=0;i<n;++i){
      ret[i]=c;
   }
   return string(ret);
}
struct primeFactors{
   vector<int> factors;
   vector<int> exponents;
}
bool isPrime(int num){
   for(int i=0;i<=sqrt(num);i++){
      if(num%i==0){
         return false;
      }
   }
   return true;
}
struct primeFactors factorize(int num){
   vector<int> factors;
   vector<int> exponents;
   int copyNum=num;
   for(int i=2;i<num-1;i++){
     
   }
  return {factors,exponents};
}

void printNum(int num, int divisor){
   string snum=to_string(num);//googled
   int slen=snum.length()+2;
   printf("%i | %i\n",num, divisor);
   for(int i=0;i<slen-1;++i){
      printf("-");
   }
   printf("+\n");
   //printf("%s","-"*(slen-1)+"+\n");
   //printf("%s",(nChar(slen-1,'-')+"+\n").c_str());
}

int main(){
   printNum(132,11);
   printNum(12,3);
   return 0;
}
