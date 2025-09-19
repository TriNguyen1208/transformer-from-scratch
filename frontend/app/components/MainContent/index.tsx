'use client'
import React from 'react'
import { OptionIcon, ShareIcon } from '../Icon'
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
                        <span>Share</span>
                    </div>
                    <div className='py-2 px-3 hover:bg-hover-button hover:rounded-lg'>
                        <div className='w-6 h-6 cursor-pointer'><OptionIcon/></div>
                    </div>
                </div>
            </div>
            <div className='text-center pt-40 font-normal text-4xl'>
                What's on the agenda today?
            </div>
            <div className='sticky bottom-4 pt-30 text-center'>
                <input
                    className='bg-[#9b9b9b4d] h-[20%] w-[60%]'
                />
            </div>
        </div>
    )
}

export default MainContent