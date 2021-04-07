#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include<string>
#include<vector>
#include<assert.h>
#include<chrono>
#include <unistd.h>
#include<iterator>
#include<cstddef>
#include <algorithm>
#define dtimer(expr) {auto start=std::chrono::high_resolution_clock::now();\
expr;\
auto stop=std::chrono::high_resolution_clock::now();\
int diff=std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();\
printf(#expr);printf("took %i nanoseconds\n",diff);}

#define do_func(expr) printf(#expr);printf(" is happening\n");
#define EXEC(expr) printf(#expr);printf(" is ");std::cout<<expr<<std::endl;
#define EXEC_VEC(expr,type) printf(#expr);printf(" is ");print_vec<type>(expr);

class slice_part{
   public:
   bool is_slice;
   int start;
   int stop;
   slice_part(int s1,int s2,int mode=true){
      is_slice=mode;
      start=s1;
      stop=s2;
      //printf("start:%i, stop:%i, is_slice:%i\n",start,stop,is_slice);
   }
   slice_part(int s1){
      start=s1;
      stop=1;
      is_slice=false;
   }
};
template <class t>
inline std::vector<t> slice_vector(std::vector<t> body,int start,int stop){
   std::vector<t> result(body.begin()+start,body.begin()+stop);
   return result;
}
template <class t>
int mul_vec(std::vector<t> src){
    int total=1;
    for(int i=0;i<src.size();i++){
       total*=src[i];
    }
    return total;
}
  
template <class t>
void print_vec(std::vector<t> a){
   for(int i=0;i<a.size();i++){
      std::cout<<a[i]<<"  ";
   }
   std::cout<<std::endl;
}
int get_new_numel(std::vector<int> orig_shape,std::vector<slice_part> idxs){
   int total=1;
   int acum=0;
   for(int i=0;i<idxs.size();i++){
      if(acum){
         if(idxs[i].is_slice){
            total*=(idxs[i].stop-idxs[i].start);
         }
      }else{
          if(idxs[i].is_slice){
             acum=1;
             total*=(idxs[i].stop-idxs[i].start);
          }
      }
  }
   return total;
}
class IdxWrap{
   public:
      slice_part item;
      IdxWrap(slice_part item_):item(item_){}
      IdxWrap(int idx):item(slice_part(idx,-1,1)){}
};

// my atttempt at copying pytorch tensoriterator; I have no idea how tensoriterator works, this is my interpretation
class vec_iterator{
   private:
   //note: this only works for indexes that are the same size as the shape of the vector
   std::vector<slice_part> idxs;//store idxs
   std::vector<int> shape;//shape of vector
   std::vector<int> strides;//strides of vector
   int n;//index getter
   std::vector<int> current_pos;//vector of current indexes
   std::vector<int> endings;//bounds for current_pos
   int strided;//bases strided
   public:
   int new_numel;//total amount of elements

   vec_iterator(std::vector<slice_part> slices,std::vector<int> shapeof){
      shape=shapeof;
      idxs=slices;
      std::vector<int> to_be_strides(shape.size());
      int cstride=0;
      std::vector<int> cpos(slices.size());
      std::vector<int> ends(slices.size());
      for (int i=0;i<shape.size();i++){
         int total=1;
         for (int j=i+1;j<shape.size();j++){
            total*=shape[j];
         }
         auto c=slices[i];
         cstride+=c.start*total;
         //maybe use ternary operator
         if(c.is_slice){
            ends[i]=c.stop;
         }else{
            //VERY IMOPORTANT: if the indexer is not a slice it has to be equal to the start position, it saves an extra check in recompute()
            ends[i]=c.start;
         }
         cpos[i]=c.start;
         to_be_strides[i]=total;
      }
      
      strides=to_be_strides;
      current_pos=cpos;
      endings=ends;
      strided=cstride;
      new_numel=get_new_numel(shapeof,slices);
      n=strided;

   }
   void recompute(){
      for(int i=current_pos.size()-1;i>-1;i--){
         if (current_pos[i]==endings[i]-1){
            int to_zero=idxs[i].start;
            strided-=(current_pos[i]-to_zero)*strides[i];
            current_pos[i]=to_zero;
         }else if(current_pos[i]<endings[i]-1){
            current_pos[i]++;
            strided+=strides[i];
            break;
         }
      }
      n=strided;
   }

   int get(){
      int index=strided; 
      this->recompute();
      //vec_iterator::recompute();
      return index;
   }
   
   int random_access(int step){
      return 0;
   }
};

class rich_vector{
   public:
      std::vector<float>& data;
      std::vector<int> stride_vec;
      std::vector<int> shape;
      bool strides_computed;
      rich_vector(std::vector<float>& items,std::vector<int> shapeof):data(items),shape(shapeof),strides_computed(false){}

      std::vector<int> strides(){
         if (strides_computed){
            return stride_vec;
         }
         std::vector<int> to_be_strides(shape.size());
         for(int i=0;i<shape.size();i++){
            int total=1;
            for(int j=i+1;j<shape.size();j++){
               total*=shape[j];
            }
            to_be_strides[i]=total;
         }
         stride_vec=to_be_strides;
         strides_computed=true;
         return stride_vec;
     }
     float& operator[](int index){
        return data[index];
     }
     class iterator{
        private:
            std::vector<float>& data;
            std::vector<int> shape;
            std::vector<int> strides;
            std::vector<slice_part> indexes;
            int n;
            int count;
            vec_iterator indexer;
        public:
            using difference_type =std::ptrdiff_t;
            using value_type =float;
            using pointer =float*;
            using reference =float&;
            using iterator_category = std::forward_iterator_tag;
            iterator(std::vector<float>& data_,std::vector<int> shape_, std::vector<int> strides_,std::vector<slice_part> indexes_):data(data_),
            shape(shape_),strides(strides_),indexes(indexes_),indexer(vec_iterator(indexes_,shape_)),count(0) {n=indexer.get();}
            float& operator*(){return data[n];}
            iterator operator++(){n=indexer.get();count++;return *this;}
            void operator++(int junk){n=indexer.get();count++;}
            bool operator==(iterator other){return count==other.indexer.new_numel;}
            bool operator!=(iterator other){return count!=other.indexer.new_numel;}
            iterator begin(){return *this;}
            iterator end(){return *this;}
     };
     iterator make_iter(std::vector<slice_part> idxs){
        return iterator(data,shape,this->strides(),idxs);
     }
};
void fill_arr_(std::vector<float>& src,float val,int length){
   for(int i=0;i<src.size();i++){
      src[i]=val;
   }
}

std::vector<float> arange(int start,int stop){
   std::vector<float> res(stop-start);
   int count=0;
   for(int i=start;i<stop;i++){
      res[count]=i;
      count+=1;
   }
   return res;
}

std::vector<float> iterate_index(std::vector<float> src,std::vector<int> shape,std::vector<slice_part> idxs){
   vec_iterator iter(idxs,shape);
   
   std::vector<float> res(iter.new_numel);
   //int index;
   for(int i=0;i<iter.new_numel;i++){
      int index=iter.get();
      res[i]=src[index];
   }
   
   return res;
}
void fill_iter(rich_vector::iterator start, rich_vector::iterator stop, float val){
   for(auto i=start;i!=stop;i++){
      *i=val;
   }
}
void plus_equal(float& thing){
   //printf("f is %f\n",thing);
   thing+1;
   return;
}
int main(){
   std::vector<int> shape={50,5,40};
   std::vector<slice_part> slices={slice_part(1,40),3,20};
   std::vector<float> arr1=arange(0,10000);
   rich_vector rv1(arr1,shape);
   auto i1=rv1.make_iter({slice_part(1,40),3,20});
   auto i2=rv1.make_iter({slice_part(1,40),3,20});
   printf("numel is %i\n",get_new_numel(shape,slices));
   printf("done\n");
   dtimer(iterate_index(rv1.data,shape,{slice_part(1,40),3,20}));
   dtimer(std::for_each(i1.begin(),i1.end(),plus_equal));
   dtimer(for(auto i:i2){plus_equal(i);})
   return 0;
}

