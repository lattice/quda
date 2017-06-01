#pragma once

namespace quda {

  inline char *i32toa(char *s, int32_t n) {
    int8_t i,j;                    //decade counter
    int8_t idx=0;                  //string index
    int32_t const Div[10] = {      //decades table
      1000000000L,100000000L,10000000L,1000000L,
      100000L,10000L,1000L,100L,10L,1};
    int32_t b;                     //i32 to hold table read

    for (i=0; i<10;i++) {        //do all the decades, start with biggest
      j=0;                       //clear the decimal digit counter
      b=Div[i];                  //read the table once;
      while (n>=b) {             //T: "left-over" still bigger then decade; substr. and count
	j++;
	n-=b;
      }
      if (j) {          //T: decade count!=0 or first digit has been detected
	s[idx++]='0'+j;          //..then add the decade count
      }
    }

    s[idx]=0;                    //end the string
    return(s+idx);               //return last written pointer
  }

}
