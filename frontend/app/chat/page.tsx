'use client'
import React from 'react'
import { OptionIcon, ShareIcon } from '../components/Icon'
const MainContent = () => {
    const handleShareButton = () => {
        console.log("Share button")
    }
    const handleClickNewChat = () => {
        console.log("Weather gpt")
    }
    return (
        <div className='h-screen bg-main w-full text-white overflow-y-auto'>
            <div className="flex flex-row justify-between py-5 px-5 sticky top-0 z-9999">
                <span 
                    className='py-2 px-3 hover:bg-hover-button hover:rounded-xl cursor-pointer'
                    onClick={handleClickNewChat}
                >
                    WeatherGPT
                </span>
                <div className='flex flex-row gap-4'>
                    <div 
                        className='flex flex-row gap-3 cursor-pointer hover:bg-hover-button hover:rounded-full py-2 px-3'
                        onClick={handleShareButton}
                    >   
                        <div className='w-6 h-6'><ShareIcon/></div>
                        <span>Chia sẻ</span>
                    </div>
                    <div className='py-2 px-3 hover:bg-hover-button hover:rounded-lg'>
                        <div className='w-6 h-6 cursor-pointer'><OptionIcon/></div>
                    </div>
                </div>
            </div>
            <div className='text-center pt-40 font-normal text-4xl'>
                Thời tiết ở Hà Nội hôm nay thế nào?
            </div>
            {/* <div className='sticky bottom-4 pt-30 text-center'>
                <input
                    className='bg-[#9b9b9b4d] h-[20%] w-[60%]'
                />
            </div> */}

            <div className="sticky bottom-0 pt-10">
                <div className="relative w-full max-w-3xl mx-auto">
                    <input
                        type="text"
                        placeholder="Đặt câu hỏi ở đây..."
                        className="w-full bg-[#9b9b9b4d] rounded-full px-4 py-4 pr-20 text-white placeholder-gray-300 outline-none"
                    />
                    <button className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-500 px-4 py-2 rounded-full text-black font-semibold transition hover:bg-blue-400">
                        Gửi
                    </button>
                </div>
            </div>
        </div>
    )
}

export default MainContent