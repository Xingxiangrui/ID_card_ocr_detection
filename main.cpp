
#include <time.h>
#include <unistd.h>
#include <stdio.h>



int main()
{
	int int_num=0xFFFF0000;
	int int_size=sizeof(int_num);
	
	printf("int_num=%d, int_size=%d\n",int_num,int_size);
	printf("int_num=%f, double_size=%d\n",int_num,sizeof(double));
	
    return 0;
}
